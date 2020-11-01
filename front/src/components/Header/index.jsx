import React from 'react';
import { withRouter } from 'react-router-dom';
import {Button, Form, FormControl, Nav, Navbar, NavDropdown} from "react-bootstrap";

function Header(props) {
  return (
    <Navbar bg="light" expand="lg">
      <Navbar.Brand href="/">Архивариус</Navbar.Brand>
      <Navbar.Toggle aria-controls="basic-navbar-nav" />
      <Navbar.Collapse id="basic-navbar-nav">
        <Nav className="mr-auto">
          <Nav.Link href="/search">Поиск</Nav.Link>
          <Nav.Link href="/upload">Загрузка</Nav.Link>
          <NavDropdown title="Отчёты" id="basic-nav-dropdown">
            <NavDropdown.Item href="/stats/system">Систестема</NavDropdown.Item>
            <NavDropdown.Item href="/stats/bi">BI</NavDropdown.Item>
          </NavDropdown>
        </Nav>
        <Form inline method={"GET"} action={"/search"} >
          <FormControl type="text" name={'text'} placeholder="Поиск по тексту" className="mr-sm-2" />
          <Button type={"submit"} variant="outline-success">Искать</Button>
        </Form>
      </Navbar.Collapse>
    </Navbar>
  );
}

export default withRouter(Header);
